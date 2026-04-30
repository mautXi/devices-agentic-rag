FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp
WORKDIR /llama.cpp
RUN cmake -B build && cmake --build build --config Release -j$(nproc) --target llama-server

EXPOSE 8080

ENTRYPOINT ["/llama.cpp/build/bin/llama-server"]
