package com.nowcoder.community.service;

import com.nowcoder.community.dao.StockMapper;
import com.nowcoder.community.dao.UserStockMapper;
import com.nowcoder.community.entity.Stock;
import com.nowcoder.community.entity.UserStock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class StockService {

    private static final Logger logger = LoggerFactory.getLogger(StockService.class);

    private static final String API_URL = "https://finnhub.io/api/v1/quote?symbol={symbol}&token={apiKey}";

    private static final String API_URL_QUOTE = "https://finnhub.io/api/v1/quote?symbol={symbol}&token={apiKey}";
    private static final String API_URL_PROFILE = "https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={apiKey}";
    private static final String API_URL_HISTORY = "https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={from}&to={to}&token={apiKey}";
    private static final String API_KEY = "cp6g309r01qm8p9l1f20cp6g309r01qm8p9l1f2g"; // 在此处替换为你的API密钥

    @Autowired
    private StockMapper stockMapper;

    @Autowired
    private UserStockMapper userStockMapper;

    public List<Stock> findAllStocks() {
        return stockMapper.selectAllStocks();
    }

    public Map<String, Object> getStockData(String symbol) {
        RestTemplate restTemplate = new RestTemplate();
        Map<String, Object> params = new HashMap<>();
        params.put("symbol", symbol);
        params.put("apiKey", API_KEY);

        // 获取股票实时数据
        Map<String, Object> quoteResponse = restTemplate.getForObject(API_URL_QUOTE, Map.class, params);
        // 获取公司简介
        Map<String, Object> profileResponse = restTemplate.getForObject(API_URL_PROFILE, Map.class, params);

        Map<String, Object> result = new HashMap<>();
        if (quoteResponse != null) {
            result.put("currentPrice", quoteResponse.get("c"));
            result.put("openPrice", quoteResponse.get("o"));
            result.put("highPrice", quoteResponse.get("h"));
            result.put("lowPrice", quoteResponse.get("l"));
            result.put("prevClosePrice", quoteResponse.get("pc"));
            result.put("priceChange", quoteResponse.get("d"));
            result.put("priceChangePercent", quoteResponse.get("dp"));
        }

        if (profileResponse != null) {
            result.put("companyName", profileResponse.get("name"));
            result.put("ticker", profileResponse.get("ticker"));
            result.put("exchange", profileResponse.get("exchange"));
            result.put("industry", profileResponse.get("finnhubIndustry"));
            result.put("website", profileResponse.get("weburl"));
            result.put("country", profileResponse.get("country"));
        }

        logger.info("Stock Data for {}: {}", symbol, result); // 添加日志输出
        return result;
    }
    public void followStock(int userId, String stockCode) {
        Stock stock = stockMapper.selectByCode(stockCode);
        if (stock != null) {
            UserStock userStock = new UserStock();
            userStock.setUserId(userId);
            userStock.setStockId(stock.getId());
            userStockMapper.insertUserStock(userStock);
        }
    }

    public void unfollowStock(int userId, String stockCode) {
        Stock stock = stockMapper.selectByCode(stockCode);
        if (stock != null) {
            userStockMapper.deleteUserStock(userId, stock.getId());
        }
    }

    public boolean isFollowing(int userId, String stockCode) {
        Stock stock = stockMapper.selectByCode(stockCode);
        return stock != null && userStockMapper.countUserStock(userId, stock.getId()) > 0;
    }

    // 获取用户关注的股票信息
    public List<UserStock> findUserStocks(int userId) {
        return userStockMapper.selectUserStocksByUserId(userId);
    }

    // 通过股票ID获取股票信息
    public Stock findStockById(int stockId) {
        return stockMapper.selectById(stockId);
    }
}
