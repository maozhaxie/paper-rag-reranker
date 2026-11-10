"""Bailian reranking; imports and offline tests never make model requests."""
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request


class RerankError(RuntimeError):
    """Safe user-facing error: never include credentials or HTTP response bodies."""


def select_ranked_sources(candidates, results, k):
    expected = min(k, len(candidates))
    if not isinstance(results, list) or len(results) != expected:
        raise RerankError("重排返回的条数异常，本次没有使用重排结果。")
    seen, ranked = set(), []
    for result in results:
        if not isinstance(result, dict):
            raise RerankError("重排结果格式异常。")
        index, score = result.get("index"), result.get("relevance_score")
        if (isinstance(index, bool) or not isinstance(index, int) or
                not 0 <= index < len(candidates) or index in seen or
                isinstance(score, bool) or not isinstance(score, (int, float)) or
                not math.isfinite(score) or not 0 <= score <= 1):
            raise RerankError("重排返回的索引或分数异常，本次没有使用重排结果。")
        seen.add(index)
        item = dict(candidates[index])
        item.update(vector_rank=item.get("vector_rank", item["id"]), rerank_score=float(score))
        ranked.append(item)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    for position, item in enumerate(ranked, 1):
        item["id"] = position
    return ranked


def rerank_sources(question, candidates, k, cfg):
    if not candidates:
        return []
    if not cfg.key or cfg.key == "replace_with_your_key":
        raise RerankError("请先在本地 .env 配置百炼密钥。")
    endpoint = urlsplit(cfg.rerank_url)
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password:
        raise RerankError("RERANK_URL 需要是不带用户名密码的 HTTPS 接口地址。")
    payload = {"model": cfg.rerank_model, "query": question,
               "documents": [item["text"] for item in candidates],
               "top_n": min(k, len(candidates)), "return_documents": False}
    req = Request(cfg.rerank_url, data=json.dumps(payload).encode("utf-8"), method="POST",
                  headers={"Authorization": "Bearer " + cfg.key, "Content-Type": "application/json"})
    try:
        # Do not forward Authorization to a redirected host.
        from urllib.request import HTTPRedirectHandler, build_opener
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        with build_opener(NoRedirect).open(req, timeout=60) as response:
            body = json.loads(response.read(2 * 1024 * 1024))
    except HTTPError as exc:
        status = exc.code
        exc.close()
        if status in (400, 401, 403, 404):
            raise RerankError(f"百炼重排调用失败（HTTP {status}）。请核对模型权限、余额、地域和 RERANK_URL；可取消勾选重排后继续使用向量检索。") from None
        raise RerankError(f"百炼重排服务暂时不可用（HTTP {status}），请稍后重试或取消勾选重排。") from None
    except (URLError, TimeoutError, OSError):
        raise RerankError("重排接口连接失败或超时，请检查网络；可取消勾选重排后继续检索。") from None
    except (ValueError, UnicodeError):
        raise RerankError("重排接口返回了无法解析的数据。") from None
    if not isinstance(body, dict):
        raise RerankError("重排接口返回格式异常。")
    return select_ranked_sources(candidates, body.get("results"), k)
