docker build -q ../ > ${IMAGE_ID}
docker run -v $PWD:/ --name transformer_dialogue --gpus all ${IMAGE_ID}