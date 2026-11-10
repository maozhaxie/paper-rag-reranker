"""Run with project Python; open Firefox at http://127.0.0.1:5000."""
from flask import Flask, jsonify, render_template, send_file
from werkzeug.exceptions import HTTPException
from api_integration import create_blueprint
from configuration import ROOT
from reranker import RerankError


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    app.register_blueprint(create_blueprint())

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/questions")
    def questions():
        return send_file(ROOT / "evaluation/questions.json", mimetype="application/json")

    @app.errorhandler(ValueError)
    def bad_input(exc):
        return jsonify(success=False, error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(success=False, error=exc.description), exc.code

    @app.errorhandler(Exception)
    def service_error(exc):
        # Avoid echoing SDK responses or keys to the browser.
        return jsonify(success=False, error=f"{type(exc).__name__}：请核对 Python 依赖、Milvus 服务和模型配置。",
                       http_status=getattr(exc, "status_code", None)), 502

    @app.errorhandler(RerankError)
    def rerank_error(exc):
        return jsonify(success=False, error=str(exc)), 502

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
