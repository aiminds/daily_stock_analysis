# -*- coding: utf-8 -*-
"""
===================================
Market Review Analysis Module
===================================

Responsibilities:
1. Fetch market index data
2. Search market news for intelligence
3. Generate daily market review report using LLM
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """Market Index Data"""
    code: str                    
    name: str                    
    current: float = 0.0         
    change: float = 0.0          
    change_pct: float = 0.0      
    open: float = 0.0            
    high: float = 0.0            
    low: float = 0.0             
    prev_close: float = 0.0      
    volume: float = 0.0          
    amount: float = 0.0          
    amplitude: float = 0.0       
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """Market Overview Data"""
    date: str                           
    indices: List[MarketIndex] = field(default_factory=list)  
    up_count: int = 0                   
    down_count: int = 0                 
    flat_count: int = 0                 
    limit_up_count: int = 0             
    limit_down_count: int = 0           
    total_amount: float = 0.0           
    
    top_sectors: List[Dict] = field(default_factory=list)     
    bottom_sectors: List[Dict] = field(default_factory=list)  


class MarketAnalyzer:
    """
    Market Analyzer
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
    ):
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def get_market_overview(self) -> MarketOverview:
        """Fetch market overview data"""
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        overview.indices = self._get_main_indices()

        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
        
        return overview

    def _get_main_indices(self) -> List[MarketIndex]:
        """Fetch main index realtime quotes"""
        indices = []

        try:
            logger.info("[Market] Fetching main index quotes...")

            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[Market] Index fetch failed, relying on news search.")
            else:
                logger.info(f"[Market] Retrieved {len(indices)} indices.")

        except Exception as e:
            logger.error(f"[Market] Index fetch error: {e}")

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """Fetch market advance/decline stats"""
        try:
            logger.info("[Market] Fetching advance/decline stats...")

            stats = self.data_manager.get_market_stats()

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(f"[Market] Up:{overview.up_count} Down:{overview.down_count} Vol:{overview.total_amount:.0f}B")

        except Exception as e:
            logger.error(f"[Market] Stats fetch error: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        """Fetch sector rankings"""
        try:
            logger.info("[Market] Fetching sector rankings...")

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

        except Exception as e:
            logger.error(f"[Market] Sector ranking fetch error: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """Search market news"""
        if not self.search_service:
            logger.warning("[Market] Search service not configured.")
            return []
        
        all_news = []
        search_queries = self.profile.news_queries
        
        try:
            logger.info("[Market] Searching market news...")
            market_name = "US market" if self.region == "us" else "Stock Market"
            
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
            
            logger.info(f"[Market] Found {len(all_news)} news items.")
            
        except Exception as e:
            logger.error(f"[Market] News search error: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """Generate market review using LLM"""
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[Market] AI analyzer unavailable, using fallback template.")
            return self._generate_template_review(overview, news)
        
        prompt = self._build_review_prompt(overview, news)
        
        logger.info("[Market] Generating review with LLM...")
        review = self.analyzer.generate_text(prompt, max_tokens=2048, temperature=0.7)

        if review:
            logger.info("[Market] Review generated successfully.")
            return self._inject_data_into_review(review, overview)
        else:
            logger.warning("[Market] LLM returned empty, using fallback template.")
            return self._generate_template_review(overview, news)
    
    def _inject_data_into_review(self, review: str, overview: MarketOverview) -> str:
        """Inject structured data tables into LLM prose."""
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)

        if stats_block:
            review = self._insert_after_section(review, r'###\s*1\. Market Summary', stats_block)

        if indices_block:
            review = self._insert_after_section(review, r'###\s*2\. Index Commentary', indices_block)

        if sector_block:
            review = self._insert_after_section(review, r'###\s*4\. Sector/Theme Highlights', sector_block)

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        import re
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            insert_pos = len(text)
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        lines = [
            f"> 📈 Up **{overview.up_count}** / Down **{overview.down_count}** / "
            f"Flat **{overview.flat_count}** | "
            f"Limit Up **{overview.limit_up_count}** / Limit Down **{overview.limit_down_count}** | "
            f"Volume **{overview.total_amount:.0f}**"
        ]
        return "\n".join(lines)

    def _build_indices_block(self, overview: MarketOverview) -> str:
        if not overview.indices:
            return ""
        lines = [
            "| Index | Last | Change | Vol |",
            "|------|------|--------|-----------|"]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_raw = idx.amount or 0.0
            if amount_raw == 0.0:
                amount_str = "N/A"
            elif amount_raw > 1e6:
                amount_str = f"{amount_raw / 1e8:.0f}"
            else:
                amount_str = f"{amount_raw:.0f}"
            lines.append(f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | {amount_str} |")
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            top = " | ".join([f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]])
            lines.append(f"> 🔥 Leading: {top}")
        if overview.bottom_sectors:
            bot = " | ".join([f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]])
            lines.append(f"> 💧 Lagging: {bot}")
        return "\n".join(lines)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])
        
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        stats_block = f"""## Market Overview
- Up: {overview.up_count} | Down: {overview.down_count} | Flat: {overview.flat_count}
- Limit up: {overview.limit_up_count} | Limit down: {overview.limit_down_count}
- Total volume: {overview.total_amount:.0f}""" if self.profile.has_market_stats else "## Market Overview\n(Market has no equivalent advance/decline stats.)"

        sector_block = f"""## Sector Performance
Leading: {top_sectors_text if top_sectors_text else "N/A"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}""" if self.profile.has_sector_rankings else "## Sector Performance\n(Sector data not available.)"

        data_no_indices_hint = "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels." if not indices_text else ""
        indices_placeholder = indices_text if indices_text else "No index data (API error)"
        news_placeholder = news_text if news_text else "No relevant news"

        return f"""You are a professional financial market analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- ALL OUTPUT MUST BE IN ENGLISH
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self.strategy.to_prompt_block()}

---

# Output Template (follow this exact structure)

## {overview.date} Market Recap

### 1. Market Summary
(2-3 sentences on overall market performance, index moves, volume)

### 2. Index Commentary
(Analyze major index moves based on the data provided.)

### 3. Fund Flows
(Interpret volume and flow implications)

### 4. Sector/Theme Highlights
(Analyze drivers behind leading/lagging sectors)

### 5. Outlook
(Short-term view based on price action and news)

### 6. Risk Alerts
(Key risks to watch)

### 7. Strategy Plan
(Provide risk-on/neutral/risk-off stance, position sizing guideline, and one invalidation trigger.)

---

Output the report content directly, no extra commentary.
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """Generate fallback report if LLM fails"""
        mood_code = self.profile.mood_index_code
        mood_index = next((idx for idx in overview.indices if idx.code == mood_code or idx.code.endswith(mood_code)), None)
        
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = "Strong Uptrend"
            elif mood_index.change_pct > 0:
                market_mood = "Slight Uptrend"
            elif mood_index.change_pct > -1:
                market_mood = "Slight Downtrend"
            else:
                market_mood = "Significant Downtrend"
        else:
            market_mood = "Sideways Consolidation"
        
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        top_text = ", ".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = ", ".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        stats_section = ""
        if self.profile.has_market_stats:
            stats_section = f"""
### 3. Advance/Decline Stats
| Metric | Value |
|------|------|
| Up | {overview.up_count} |
| Down | {overview.down_count} |
| Limit Up | {overview.limit_up_count} |
| Limit Down | {overview.limit_down_count} |
| Volume | {overview.total_amount:.0f} |
"""
        sector_section = ""
        if self.profile.has_sector_rankings and (top_text or bottom_text):
            sector_section = f"""
### 4. Sector Performance
- **Leading**: {top_text}
- **Lagging**: {bottom_text}
"""
        strategy_summary = self.strategy.to_markdown_block()
        report = f"""## {overview.date} Market Recap

### 1. Market Summary
Today's market showed a **{market_mood}**.

### 2. Major Indices
{indices_text}
{stats_section}
{sector_section}
### 5. Risk Alerts
Investing carries risks. Data is for reference only and does not constitute financial advice.

{strategy_summary}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self) -> str:
        logger.info("========== Starting Market Review ==========")
        overview = self.get_market_overview()
        news = self.search_market_news()
        report = self.generate_market_review(overview, news)
        logger.info("========== Market Review Complete ==========")
        return report

# Test entry
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s')
    analyzer = MarketAnalyzer()
    overview = analyzer.get_market_overview()
    print(f"\n=== Market Overview ===")
    print(f"Date: {overview.date}")
    print(f"Indices count: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"Up: {overview.up_count} | Down: {overview.down_count}")
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== Review Report ===")
    print(report)
