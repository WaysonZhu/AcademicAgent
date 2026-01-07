import json
from typing import List, Dict
import os
import time
import requests
from langchain_core.tools import tool

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("semantic_tools")


class SemanticScholarAPI:
    """可以直接调用的内部帮助类，处理 HTTP 请求"""

    @staticmethod
    def _get_headers():
        # 如果有API Key则添加，没有则留空（部分SS接口可能无需key，但文档中提供了key配置）
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        if settings.AI4SCHOLAR_API_KEY:
            headers['Authorization'] = f'Bearer {settings.AI4SCHOLAR_API_KEY}'
        return headers

    @staticmethod
    def search_papers(query: str, limit: int = 10, offset: int = 0) -> List[Dict]:
        url = f"{settings.API_BASE_URL}/search"
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            # 根据需求文档返回示例，通常不需要额外指定fields，但为了保险起见，
            # 若API支持，最好指定需要 abstract, title, year, citationCount 等
            "fields": "title,authors,year,abstract,citationCount,venue,openAccessPdf,url,referenceCount,influentialCitationCount,publicationDate"
        }
        try:
            response = requests.get(url, params=params, headers=SemanticScholarAPI._get_headers())
            response.raise_for_status()
            data = response.json()

            # 将搜索结果保存到 JSON 文件
            try:
                if not os.path.exists(settings.DATA_DIR):
                    os.makedirs(settings.DATA_DIR)

                timestamp = int(time.time())
                # 文件名区分 search_papers
                file_name = f"search_papers_{timestamp}.json"
                file_path = os.path.join(settings.DATA_DIR, file_name)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                logger.info(f"[Debug] Search results saved to: {file_path}")
            except Exception as save_err:
                logger.error(f"[Debug] Failed to save search results: {save_err}")

            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []

    @staticmethod
    def get_batch_details(paper_ids: List[str]) -> List[Dict]:
        """
        批量获取论文详情。
        """
        url = f"{settings.API_BASE_URL}/batch"
        # 显式请求引用和被引用字段
        params = {
            "fields": "title,authors,year,abstract,citationCount,venue,openAccessPdf,url,referenceCount,influentialCitationCount,publicationDate,citations,references"
        }
        # Body 中包含 ids
        payload = {"ids": paper_ids}

        try:
            response = requests.post(url, params=params, json=payload, headers=SemanticScholarAPI._get_headers())
            response.raise_for_status()

            result = response.json()

            # 将 API返回结果保存到JSON文件
            try:
                # 确保 data 目录存在 (settings.DATA_DIR 已经在 config 中定义)
                if not os.path.exists(settings.DATA_DIR):
                    os.makedirs(settings.DATA_DIR)

                # 生成带有时间戳的文件名，防止覆盖
                timestamp = int(time.time())
                file_name = f"batch_details_{timestamp}.json"
                file_path = os.path.join(settings.DATA_DIR, file_name)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)

                logger.info(f"[Debug] Batch details saved to: {file_path}")
            except Exception as save_err:
                logger.error(f"[Debug] Failed to save batch details: {save_err}")

            if isinstance(result, list):
                return result
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Error getting batch details: {e}")
            return []


# --- LangChain Tools ---

@tool
def tool_search_by_keyword(query: str, limit: int = 10) -> List[Dict]:
    """
    根据关键字检索论文 (Initial Retrieval)。
    Args:
        query: 优化后的搜索关键词，例如 '"AI Agent" OR "LLM Agent"'
        limit: 返回数量，默认为10
    Returns:
        包含论文基础信息的列表 (JSON格式)
    """
    logger.info(f"Executing tool_search_by_keyword with query: {query}")
    return SemanticScholarAPI.search_papers(query, limit=limit)


@tool
def tool_search_batch_details(paper_ids: List[str]) -> List[Dict]:
    """
    根据多个论文ID批量检索详细信息 (含参考文献和引用文献)。
    用于构建引用图谱。
    Args:
        paper_ids: 论文ID列表，例如 ['id1', 'id2']
    Returns:
        包含详细信息(含 references 和 citations)的论文列表
    """
    logger.info(f"Executing tool_search_batch_details for {len(paper_ids)} papers")
    if not paper_ids:
        return []

    return SemanticScholarAPI.get_batch_details(paper_ids)


@tool
def tool_search_by_title(title: str) -> Dict:
    """
    根据论文标题精确检索论文。
    """
    url = f"{settings.API_BASE_URL}/search/match"
    params = {"query": title}
    try:
        response = requests.get(url, params=params, headers=SemanticScholarAPI._get_headers())
        response.raise_for_status()
        data = response.json().get("data", [])
        return data[0] if data else {}
    except Exception as e:
        logger.error(f"Error searching by title: {e}")
        return {}