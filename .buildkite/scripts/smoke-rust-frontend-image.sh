#!/usr/bin/env bash
# Exercise the installed binary, the named-volume permissions and native codecs.
set -euo pipefail
image=${1:?usage: smoke-rust-frontend-image.sh IMAGE [MODEL_DIRECTORY]}
work=$(mktemp -d)
cache=$(docker volume create)
container=
cleanup() {
    if [[ -n $container ]]; then
        docker logs "$container" >&2
        docker rm -f "$container" >/dev/null
    fi
    docker volume rm "$cache" >/dev/null
    rm -rf "$work"
}
trap cleanup EXIT

model_dir=${2:-$work/model}
if [[ $# == 1 ]]; then
    mkdir -p "$model_dir"
    revision=2fc06364715b967f1860aea9cf38778875588b17
    for file in config.json tokenizer.json tokenizer_config.json chat_template.jinja \
        preprocessor_config.json video_preprocessor_config.json; do
        curl -fsSL --retry 3 "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/$revision/$file" -o "$model_dir/$file"
    done
fi
container=$(docker run -d -p 127.0.0.1::8000 -v "$model_dir:/model:ro" \
    -v "$cache:/home/vllm/.cache/huggingface" "$image" \
    render /model --host 0.0.0.0 --max-model-len 8192)
docker exec "$container" sh -ec '
    test "$(id -u):$(id -g)" = 2000:0
    touch "$HOME/.cache/huggingface/.write-check"
    ldd /usr/local/bin/vllm-rs > /tmp/linked
    ! grep -q "not found" /tmp/linked || exit 1
    ffmpeg -loglevel error -f lavfi -i color=blue:s=64x64:r=2 -frames:v 4 /tmp/test.mp4
    ffmpeg -loglevel error -i /tmp/test.mp4 -frames:v 1 /tmp/test.jpg
    test "$(ffprobe -v error -select_streams v -show_entries stream=nb_frames -of csv=p=0 /tmp/test.mp4)" = 4
    ffmpeg -loglevel error -f lavfi -i sine=duration=0.1 /tmp/test.aac
    ffmpeg -loglevel error -i /tmp/test.aac -f f32le /tmp/audio.pcm
    test -s /tmp/audio.pcm
'
url="http://$(docker port "$container" 8000/tcp)"
curl -fsS --retry 30 --retry-all-errors --retry-delay 1 "$url/health"
curl -fsS "$url/v1/completions/render" -H 'Content-Type: application/json' \
    -d '{"model":"/model","prompt":"Hello"}' | jq -e '.[0].token_ids == [9419]'

for kind in image video; do
    case $kind in
        image) file=test.jpg; mime=image/jpeg ;;
        video) file=test.mp4; mime=video/mp4 ;;
    esac
    docker exec "$container" base64 -w0 "/tmp/$file" | jq -Rs \
        --arg kind "${kind}_url" --arg mime "$mime" '
        {model:"/model", messages:[{role:"user",content:[
          {type:$kind, ($kind): {url:("data:"+$mime+";base64,"+.)}},
          {type:"text",text:"Describe."}]}]}' > "$work/request.json"
    curl -fsS "$url/v1/chat/completions/render" -H 'Content-Type: application/json' \
        --data-binary "@$work/request.json" | jq -e --arg kind "$kind" '
          (.features.kwargs_data[$kind][0] | length) > 0 and
          .features.mm_placeholders[$kind][0].length > 0 and
          ([.token_ids[] | select(. == 248053)] | length) ==
            (if $kind == "video" then 3 else 1 end)'
done
