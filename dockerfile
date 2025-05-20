FROM gcc:12 AS cpp-builder

WORKDIR /build

COPY ./cl/LexerParser/include /build/include
COPY ./cl/LexerParser/src /build/src
COPY ./cl/LexerParser/api.cpp /build/

RUN g++ -std=c++17 -fPIC -shared \
    -o chronolang.so \
    api.cpp src/lexer.cpp src/parser.cpp src/ast_visualize.cpp src/astToJson.cpp \
    -Iinclude && \
    strip chronolang.so

FROM python:3.9-slim AS python-builder

WORKDIR /app
COPY cl/Interpreter/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM eclipse-temurin:21-jdk-jammy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=cpp-builder /build/chronolang.so /app/cl/Interpreter/
COPY --from=python-builder /root/.local /root/.local
COPY cl/Interpreter /app/cl/Interpreter
COPY ./target/ChronoLangServer-0.0.1-SNAPSHOT.jar app.jar

ENV PATH="/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/cl/Interpreter"
ENV PYTHONPATH="/app/cl/Interpreter"

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]