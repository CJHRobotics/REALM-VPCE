# Webots R2025a for GAIVI.
#
# Built by USF's container builder service, which converts this Dockerfile
# into a Singularity image (GAIVI runs singularity 3.6.3 and does not allow
# unprivileged `singularity build`):
#
#   container-builder-client -uwd container/Dockerfile webots_r2025a.sif
#
# The image carries Webots, Python 3.11, and a SOFTWARE OpenGL stack (Mesa
# llvmpipe). The HamBot camera is 224x224 and the lidar is 360x1, so no GPU
# is needed to render them -- which keeps jobs off the contended gpu
# partitions entirely.
#
# The REALM repo and its venv deliberately stay OUTSIDE the image and are
# bind-mounted at run time, so editing code never means rebuilding.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Python 3.11 (jammy ships 3.10; REALM pins 3.11) via deadsnakes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget gnupg software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Xvfb supplies the X display Webots' Qt front end expects; the mesa
# packages supply the llvmpipe software rasterizer behind it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb xauth \
        libgl1-mesa-dri libgl1-mesa-glx libglu1-mesa mesa-utils \
        libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
        libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xkb1 \
        libdbus-1-3 libnss3 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Webots R2025a. `apt-get install ./file.deb` resolves the .deb's own
# dependency tree, which is long and version-specific -- do not replace
# this with `dpkg -i`.
#
# VERIFY THIS URL before the first build; the release asset naming has
# shifted between Webots releases and a 404 here is the most likely
# reason for a failed build.
ARG WEBOTS_VERSION=R2025a
ARG WEBOTS_DEB=webots_2025a_amd64.deb
RUN wget -q "https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/${WEBOTS_DEB}" -O /tmp/webots.deb \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/webots.deb \
    && rm -f /tmp/webots.deb && rm -rf /var/lib/apt/lists/*

ENV WEBOTS_HOME=/usr/local/webots
ENV PATH="${WEBOTS_HOME}:${PATH}"
ENV LD_LIBRARY_PATH="${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${WEBOTS_HOME}/lib/controller/python:${PYTHONPATH}"

# Force software rendering and keep Webots from trying to phone home or
# pop dialogs on a node with no user attached.
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=llvmpipe
ENV QT_X11_NO_MITSHM=1
ENV WEBOTS_DISABLE_SAVE_SCREEN_PERSPECTIVE_ON_CLOSE=true

CMD ["/bin/bash"]
