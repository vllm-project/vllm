docker compose build --no-cache
docker compose --profile check up 
docker compose --profile pct up 

[Shell]
docker compose run --rm intel-speed-select-shell


[SST Directly]
docker compose run --rm intel-speed-select intel-speed-select core-power info

