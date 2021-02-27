docker build $PWD > ${IMAGE_ID}
docker run -v $PWD:/ --name transformer_dialogue --gpus all ${IMAGE_ID}

docker run -it -v $PWD:/transformer/ --name transformer_dialogue --gpus all test
docker exec -it 7c4e29af7db8 bash
docker exec -it transformer_dialogue bash