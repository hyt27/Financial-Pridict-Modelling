package com.nowcoder.community.controller;

import com.nowcoder.community.entity.Stock;
import com.nowcoder.community.entity.UserStock;
import com.nowcoder.community.service.StockService;
import com.nowcoder.community.entity.User;
import com.nowcoder.community.util.HostHolder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
public class StockController {

    @Autowired
    private StockService stockService;

    @Autowired
    private HostHolder hostHolder;

    @GetMapping("/stock")
    public String getStockPage(Model model) {
        return "site/stock";
    }

    @GetMapping("/stock/data")
    @ResponseBody
    public List<Map<String, Object>> getStockData() {
        List<Stock> stocks = stockService.findAllStocks();
        List<Map<String, Object>> stockData = new ArrayList<>();

        for (Stock stock : stocks) {
            Map<String, Object> stockInfo = stockService.getStockData(stock.getCode());
            if (!stockInfo.isEmpty()) {
                stockData.add(stockInfo);
            }
        }
        return stockData;
    }

    @PostMapping("/stock/follow")
    @ResponseBody
    public Map<String, Object> followStock(@RequestBody Map<String, String> request) {
        Map<String, Object> response = new HashMap<>();
        User user = hostHolder.getUser();
        if (user == null) {
            response.put("status", "error");
            response.put("message", "你还没有登录哦!");
            return response;
        }

        String stockCode = request.get("stockCode");
        stockService.followStock(user.getId(), stockCode);
        response.put("status", "success");
        response.put("message", "关注成功");
        return response;
    }

    @PostMapping("/stock/unfollow")
    @ResponseBody
    public Map<String, Object> unfollowStock(@RequestBody Map<String, String> request) {
        Map<String, Object> response = new HashMap<>();
        User user = hostHolder.getUser();
        if (user == null) {
            response.put("status", "error");
            response.put("message", "你还没有登录哦!");
            return response;
        }

        String stockCode = request.get("stockCode");
        stockService.unfollowStock(user.getId(), stockCode);
        response.put("status", "success");
        response.put("message", "取消关注成功");
        return response;
    }

    @GetMapping("/stock/isFollowing")
    @ResponseBody
    public boolean isFollowing(@RequestParam String stockCode) {
        User user = hostHolder.getUser();
        if (user == null) {
            return false;
        }
        return stockService.isFollowing(user.getId(), stockCode);
    }

    @GetMapping("/stock/my-following")
    @ResponseBody
    public List<Map<String, Object>> getMyFollowingStockData() {
        User user = hostHolder.getUser();
        if (user == null) {
            return new ArrayList<>(); // 返回空列表表示未登录用户
        }

        List<UserStock> userStocks = stockService.findUserStocks(user.getId());
        List<Map<String, Object>> stockData = new ArrayList<>();

        for (UserStock userStock : userStocks) {
            Stock stock = stockService.findStockById(userStock.getStockId());
            if (stock != null) {
                Map<String, Object> stockInfo = stockService.getStockData(stock.getCode());
                if (!stockInfo.isEmpty()) {
                    stockData.add(stockInfo);
                }
            }
        }
        return stockData;
    }

}
