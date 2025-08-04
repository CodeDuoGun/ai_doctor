#!/bin/bash
gunicorn app.main:app -w 1 -b 0.0.0.0:5001 --timeout 120 --worker-class gthread --threads 4 #--max-requests 1000 --max-requests-jitter 100