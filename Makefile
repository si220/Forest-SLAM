CONTAINER_NAME := ForestSLAM
TAG_NAME := slam
IMAGE_NAME := forest
CURRENT_DIR := $(shell pwd)

stop:
	@docker stop ${CONTAINER_NAME} || true
	@docker rm ${CONTAINER_NAME} || true

build: stop
	@docker build --tag=${IMAGE_NAME}:${TAG_NAME} .

run:
	@xhost +si:localuser:root >> /dev/null
	@docker run \
		-it \
		--privileged \
		--network host \
		--ipc host \
		--gpus all \
		-e DISPLAY=host.docker.internal:0.0 \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-e "XAUTHORITY=$XAUTH" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v ${CURRENT_DIR}/ros_ws/:/root/ros_ws/ \
		--name ${CONTAINER_NAME} \
		${IMAGE_NAME}:${TAG_NAME} \
		/bin/bash -c "source /opt/ros/noetic/setup.bash \
		&& cd /root/ros_ws \
		&& source devel/setup.bash && rosdep install \
		--from-paths src --ignore-src --rosdistro \
		noetic -y && bash"

exec:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	"source /opt/ros/noetic/setup.bash && \
	cd /root/ros_ws && \
	source devel/setup.bash && rosdep install \
	--from-paths src --ignore-src --rosdistro \
	noetic -y && bash"