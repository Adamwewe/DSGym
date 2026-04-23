FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-dev \
    bedtools \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# setuptools>=82 drops pkg_resources; pybedtools 0.9.1 setup.py imports pkg_resources
RUN pip install -U pip && \
    printf '%s\n' 'setuptools>=69,<82' > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt 'setuptools>=69,<82' wheel

COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^[[:space:]]*pybedtools' /tmp/requirements.txt > /tmp/requirements-without-pybedtools.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements-without-pybedtools.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt 'setuptools>=69,<82' wheel && \
    pip install --no-cache-dir -c /tmp/constraints.txt --no-build-isolation 'pybedtools==0.9.1'

# Copy the application code
COPY . /app
WORKDIR /app

# Set up environment
ENV PORT=8432
ENV CODE_FOLDER=""
ENV PYTHONWARNINGS="ignore::FutureWarning"

# Run the application
CMD ["python", "main.py"] 