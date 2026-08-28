default:
    just --list

test:
    go test -count 10 -shuffle on -race -v ./...

fuzz:
    go test -race -v -fuzz=Fuzz -fuzztime=60s ./...

lint:
    golangci-lint run

fmt:
    golangci-lint fmt
