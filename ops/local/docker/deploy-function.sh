sdk use java 21.0.9-librca
./mvnw -pl libraries/shared-domain -am -DskipTests install
./mvnw -DskipTests package

docker cp functions/inference-function/target/inference-function-0.0.1-SNAPSHOT.jar gf-locator:/data
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "deploy --jar=/data/inference-function-0.0.1-SNAPSHOT.jar"
docker exec -it  gf-locator  gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list functions"

 java -jar target/inference-app-0.0.1-SNAPSHOT.jar

# docker exec -it  gf-locator  gfs
# connect
# show metrics
# show metrics --region=/ProductReview
# query --query="select * from /ProductReviews"
# query --query="select key from /ProductReviews.entries"
# query --query="select key,value from /ProductReviews.entries"


query --query="select key,value from /SentimentResults.entries"
query --query="select key,value from /ProductReviews.entries"

# Works without PDX:
# query --query="select key,value from /ProductReviews.entries"
# query --query="select key,value.productName,value.review from /ProductReviews.entries"
# query --query="select value.review from /ProductReviews.entries where value.productName='Lawnmower'"

# show log --member=server1 --lines=100