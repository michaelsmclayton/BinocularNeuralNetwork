# sh ./runDeepLearningDocker.sh

# Define Docker ID
export DOCKERID=michaelsmclayton

# Define Docker container name
export CONTAINERNAME=tensorflow-training-image

# Create a new Docker image
docker image build --tag $CONTAINERNAME .

# Start BASH
docker run -it $CONTAINERNAME bash

# To stop and remove all docker containers
<<COMMENT
    docker stop $(docker ps -aq)
    docker rm $(docker ps -aq)
COMMENT

# To copy data out of container
<<COMMENT
    docker cp <containerId>:/file/path/within/container /host/path/target
    docker cp 0e3449796eb4:/deep-learning-notes/notes/code/bestLogisticRegressionModel.pkl /Users/michaelclayton/Documents/DeepLearning/deep-learning-notes/notes/code
COMMENT

# python-tk