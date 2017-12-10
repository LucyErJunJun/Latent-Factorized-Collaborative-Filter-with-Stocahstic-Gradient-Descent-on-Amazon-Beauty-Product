## Using the Api

### Testing the api

 curl -X GET http://localhost:5000/api/v1/recommend/abc123
 curl -X GET http://localhost:5000/api/v1/price/predict/100

The above will give an authorization error. To fix this create
an user first by submitting the following request:

### Create a token

curl -H "Content-Type: application/json" -X POST -d '{"username":"AZXP46IB63PU8","password":"abc123"}' http://localhost:5000/auth

This will create a unique access token:


you can now access the endpoint using the generated token:


curl -X GET http://localhost:5000/api/v1/recommend/abc123  -H "Authorization: JWT  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE1MTI5MDY1ODUsImlhdCI6MTUxMjkwNjU4NSwiZXhwIjoxNTEzNTExMzg1LCJpZGVudGl0eSI6MTIzfQ.JHWfKA6BWhwy0hIpNm9UvC40B22KUO_z-fuaLQvfywM""

curl -X GET http://localhost:5000/api/v1/price/predict/100  -H "Authorization: JWT  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE1MTI5MDY1ODUsImlhdCI6MTUxMjkwNjU4NSwiZXhwIjoxNTEzNTExMzg1LCJpZGVudGl0eSI6MTIzfQ.JHWfKA6BWhwy0hIpNm9UvC40B22KUO_z-fuaLQvfywM""
