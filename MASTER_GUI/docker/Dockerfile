FROM python:3.9-slim-bookworm

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=bookworm
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y \
    sudo \
    git \
    python3-tk \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    x11-utils \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Yiyu: Install packages ========================================

# Switch from root to user
USER $USERNAME

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# bashrc: Source the ROS setup file and use color prompt
RUN sed -i -e 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc