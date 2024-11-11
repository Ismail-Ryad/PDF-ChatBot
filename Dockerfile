#Base-lightweight python image
FROM python:3.10.0-slim as base

#set base working directory
WORKDIR /pdf_chatbot


# Download dependencies 
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8501
