# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# demo_spider/pipelines.py
import json
from datetime import datetime

class SortByDateJsonPipeline:
    def __init__(self, output_path):
        self.output_path = output_path
        self._items = []

    @classmethod
    def from_crawler(cls, crawler):
        path = crawler.settings.get("SORTED_JSON_OUTPUT", "ttu_cse_news.sorted.json")
        return cls(output_path=path)

    def open_spider(self, spider):
        self._items.clear()

    def process_item(self, item, spider):
        self._items.append(dict(item))
        return item

    def close_spider(self, spider):
        def key(it):
            s = it.get("published_at") or "1900-01-01"
            try:
                return datetime.strptime(s, "%Y-%m-%d")
            except:
                return datetime.min
        with_dates = [x for x in self._items if x.get("published_at")]
        without = [x for x in self._items if not x.get("published_at")]
        with_dates.sort(key=key, reverse=True)   # 由新到舊
        data = with_dates + without
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
