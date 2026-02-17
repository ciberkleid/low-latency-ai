# Access endpoints to exercise client-side and server-side inference

# Swagger UI:
# open http://localhost:8080/swagger-ui/index.html

echo
echo "### Client-side inference"
curl -X 'POST' \
  'http://localhost:8080/ai/inference/checkSentiment' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d 'I love Spring'

echo
echo
echo "### Server-side inference via function - request 1"
curl -X 'GET' \
  'http://localhost:8080/product/review/Lawnmower' \
  -H 'accept: */*'

echo
echo
echo "### Server-side inference via function - request 2"
curl -X 'GET' \
  'http://localhost:8080/product/review/Coffee%20Mug' \
  -H 'accept: */*'
echo
echo