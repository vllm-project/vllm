# vllm-completion.template.bash
# 
# To help auto-generate vllm-completion.bash
#
#############################################

_vllm_completions(){
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local subcommands={{ subcommands }}

    # Nested subcommand
    local bench_subcommands={{ bench_subcommands }}

    # Subcommands args
    local chat_args={{ chat_args }}

    local complete_args={{ complete_args }}

    local serve_args={{ serve_args }}

    local run_batch_args={{ run_batch_args }}

    local bench_latency_args={{ bench_latency_args }}

    local bench_serve_args={{ bench_serve_args }}

    local bench_throughput_args={{ bench_throughput_args }}

    # Option value completion mapping (centralized definition)
    declare -A option_value_map=(
        {{ option_value_map_entries }}
    )

    if [[ -n "${option_value_map[$prev]}" ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${option_value_map[$prev]}" -- "$cur")
        return 0
    fi

    # Top-level subcommands
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${subcommands}" -- "$cur")
        return 0
    fi

    # Second-level handling
    case "${COMP_WORDS[1]}" in
        bench)
            if [[ ${COMP_CWORD} -eq 2 ]]; then
                mapfile -t COMPREPLY < <(compgen -W "${bench_subcommands}" -- "$cur")
                return 0
            fi

            # bench subcommands （latency, serve, throughput）
            local bench_sub=${COMP_WORDS[2]}
            case "$bench_sub" in
                latency)
                    mapfile -t COMPREPLY < <(compgen -W "${bench_latency_args}" -- "$cur")
                    ;;
                serve)
                    mapfile -t COMPREPLY < <(compgen -W "${bench_serve_args}" -- "$cur")
                    ;;
                throughput)
                    mapfile -t COMPREPLY < <(compgen -W "${bench_throughput_args}" -- "$cur")
                    ;;
            esac
            return 0
            ;;
        chat)
            mapfile -t COMPREPLY < <(compgen -W "${chat_args}" -- "$cur")
            return 0
            ;;
        complete)
            mapfile -t COMPREPLY < <(compgen -W "${complete_args}" -- "$cur")
            return 0
            ;;
        serve)
            mapfile -t COMPREPLY < <(compgen -W "${serve_args}" -- "$cur")
            return 0
            ;;
        run-batch)
            mapfile -t COMPREPLY < <(compgen -W "${run_batch_args}" -- "$cur")
            return 0
            ;;
        *)
            ;;
    esac
}

complete -F _vllm_completions vllm
