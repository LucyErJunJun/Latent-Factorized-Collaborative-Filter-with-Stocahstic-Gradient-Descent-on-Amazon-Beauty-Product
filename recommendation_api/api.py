from flask import Flask
from flask_restful import Resource, Api
from flask_jwt import JWT, jwt_required
import datetime
from recommendation import Recommender

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gfe_rd=cr&dcr=0&ei=ImotWvyjD9SK8Qfp3bDYAw'
app.config['JWT_EXPIRATION_DELTA'] = datetime.timedelta(weeks=1)

api = Api(app, prefix="/api/v1")

USER_DATA = {
    "AZXP46IB63PU8": "abc123",
    "ryan":"abc123",
    "lucy":"abc123"
}

class User(object):
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return "User(id='%s')" % self.id


def verify(username, password):
    if not (username and password):
        return False
    if USER_DATA.get(username) == password:
        return User(id=123)


def identity(payload):
    user_id = payload['identity']
    return {"user_id": user_id}


jwt = JWT(app, verify, identity)


class ProductRecommenderResource(Resource):
    @jwt_required()
    def get(self, user_id):
        r = Recommender()
        recommendations_list  = r.recommend(user_id)
        return recommendations_list

api.add_resource(ProductRecommenderResource, '/recommend/<user_id>')


if __name__ == '__main__':
    app.run(debug=True)

