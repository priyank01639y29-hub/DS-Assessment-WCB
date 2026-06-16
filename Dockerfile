FROM python:3.12-slim

# PyMuPDF and psycopg[binary] ship manylinux wheels, so no build toolchain is
# needed. git is kept only so you can optionally clone VectifyAI/PageIndex (§3.2).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 8888 = JupyterLab, 8501 = optional Streamlit UI
EXPOSE 8888 8501

# Token-less for local convenience. See DOCKER.md before exposing beyond localhost.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/workspace"]
