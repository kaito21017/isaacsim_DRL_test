ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
FROM ${BASE_IMAGE}

ARG ISAACLAB_PATH=/workspace/isaaclab
ENV ISAACLAB_PATH=${ISAACLAB_PATH}
ENV ISAACLAB_SH=${ISAACLAB_PATH}/isaaclab.sh
ENV APP_DIR=/workspace/isaacsim_DRL_test
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]
WORKDIR ${APP_DIR}

COPY . ${APP_DIR}

RUN chmod +x ${APP_DIR}/isaacsim.sh && \
    ${ISAACLAB_SH} -p -m pip install --no-cache-dir -r ${APP_DIR}/requirements.txt

ENTRYPOINT ["./isaacsim.sh"]
CMD ["-p", "scripts/train_upright_policy.py", "--help"]
