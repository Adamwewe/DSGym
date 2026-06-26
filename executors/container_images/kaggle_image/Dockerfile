FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/local/bin/pip /usr/bin/pip


# Accept UID/GID as build arguments (default to 1000 for standard Linux user)
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a non-root user with matching UID/GID to host user
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -r -u ${USER_ID} -g appuser appuser

# Copy the executor application code
COPY main.py kernel_executor.py instruction.txt output_cleaning.py /app/
WORKDIR /app

# Set up environment variables
ENV PORT=8432
ENV CODE_FOLDER="/data"
ENV PYTHONWARNINGS="ignore::FutureWarning"

# Expose the port
EXPOSE 8432

# Install dependencies once during build (not at runtime)
COPY requirements.kaggle.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create submission directory and set ownership
RUN mkdir -p /submission && chown -R appuser:appuser /submission /app
RUN chmod -R 777 /submission /app

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "main.py"]