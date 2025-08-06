#!/bin/bash

# Wait for the database to be ready
while ! curl -f http://localhost:5001/ping; do
    echo "Waiting for chat service to be ready..."
    sleep 2
done

# Start the Flask application
exec python app.py