FROM redis:7.2

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-full \
    python3-pip \
    python3-setuptools \
    python3-venv \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone and build RediSearch
RUN git clone --recursive https://github.com/RediSearch/RediSearch.git /tmp/RediSearch \
    && cd /tmp/RediSearch \
    && make setup \
    && make build \
    && mkdir -p /usr/lib/redis/modules \
    && cp bin/linux-arm64-release/search/redisearch.so /usr/lib/redis/modules/ \
    && rm -rf /tmp/RediSearch

# Set up Redis configuration
COPY redis.conf /usr/local/etc/redis/redis.conf

CMD ["redis-server", "/usr/local/etc/redis/redis.conf"] 