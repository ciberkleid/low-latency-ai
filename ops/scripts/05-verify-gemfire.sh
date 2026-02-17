# docker exec -it gf-locator gfsh
# connect

# show log --member=server1 --lines=100
# list regions
# list functions

# show metrics
# show metrics --region=/ProductReview

# query --query="select * from /ProductReviews"
# query --query="select key from /ProductReviews.entries"
# query --query="select key,value from /ProductReviews.entries"

docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list functions"
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list regions"
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "query --query='select key,value from /SentimentResults.entries'"
# docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "query --query='select key,value from /ProductReviews.entries'"

### Works without PDX:
### query --query="select key,value from /ProductReviews.entries"
### query --query="select key,value.productName,value.review from /ProductReviews.entries where value.productName='Lawnmower'"