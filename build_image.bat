@ECHO OFF

SET image_name=%2
SET image_tag=latest
SET full_image_name=%image_name%:%image_tag%

docker build %1 -t %full_image_name%
docker push %full_image_name%

docker inspect --format="{{index .RepoDigests 0}}" "%full_image_name%"
