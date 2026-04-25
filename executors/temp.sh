python generate_compose.py -n 64 --types "executor-bio:64" -m ../data/data --output docker-compose-bio.yml -c container_config_bio.json

docker build -t executor-bio ./container_images/dsbio_image
docker build -t manager-prebuilt ./manager
docker-compose -f docker-compose-bio.yml up -d --build

