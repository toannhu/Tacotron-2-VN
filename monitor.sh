#!/bin/bash

myscript(){
    python3 demo_server.py
}

until myscript; do
    echo "'demo_server.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done