## Using the Api

### Testing the api

 curl -X GET http://localhost:5000/api/v1/recommend/abc123


 curl -X GET http://localhost:5000/api/v1/price/predict/100

The above will give an authorization error. To fix this create
an user first by submitting the following request:

### Create a token

curl -H "Content-Type: application/json" -X POST -d '{"username":"ryan","password":"abc123"}' http://localhost:5000/auth

This will create a unique access token:


you can now access the endpoint using the generated token:


curl -X GET http://localhost:5000/api/v1/recommend/AZXP46IB63PU8  -H "Authorization: JWT  "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJuYmYiOjE1MTI5MjYwMTYsImV4cCI6MTUxMzUzMDgxNiwiaWF0IjoxNTEyOTI2MDE2LCJpZGVudGl0eSI6MTIzfQ.9g8NK4hHYZOyK64CjhB8LWVjc93GWHY-J\nQNDDUBe5A""

curl -X GET http://localhost:5000/api/v1/price/predict/100  -H "Authorization: JWT  "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJuYmYiOjE1MTI5MjYwMTYsImV4cCI6MTUxMzUzMDgxNiwiaWF0IjoxNTEyOTI2MDE2LCJpZGVudGl0eSI6MTIzfQ.9g8NK4hHYZOyK64CjhB8LWVjc93GWHY-J\nQNDDUBe5A""
