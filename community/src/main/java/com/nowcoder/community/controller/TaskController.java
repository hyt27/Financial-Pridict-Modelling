package com.nowcoder.community.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nowcoder.community.annotation.LoginRequired;
import com.nowcoder.community.dao.TaskMapper;
import com.nowcoder.community.entity.Page;
import com.nowcoder.community.entity.Task;
import com.nowcoder.community.entity.User;
import com.nowcoder.community.service.TaskService;
import com.nowcoder.community.util.CommunityUtil;
import com.nowcoder.community.util.HostHolder;
import com.sun.xml.bind.v2.TODO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;

@Controller
public class TaskController {
    @Autowired
    private HostHolder hostHolder;
    @Autowired
    private TaskService taskService;
    @Autowired
    private TaskMapper taskMapper;

    @LoginRequired
    @RequestMapping(path = "/predict", method = RequestMethod.GET)
    public String getTaskList(Model model, Page page) {
        User user = hostHolder.getUser();
        // 分页信息
        page.setLimit(5);
        page.setPath("/predict");
        page.setRows(taskService.findTaskCount(user.getId()));
        // 任务列表
        List<Task> taskList = taskService.findTasksByUserId(user.getId(), page.getOffset(), page.getLimit());
        List<Map<String, Object>> tasks = new ArrayList<>();
        if (taskList != null) {
            for (Task task : taskList) {
                Map<String, Object> map = new HashMap<>();
                map.put("id", task.getId());  //任务ID
                map.put("category", task.getCategory());//任务种类
                map.put("stock", task.getStock());  // 股票ID
                map.put("startDate", task.getStartDate());  // 开始日期
                map.put("endDate", task.getEndDate());  // 结束日期
                map.put("userid", task.getUserid());
                map.put("model", task.getModel());  // 模型名称
                map.put("strategy", task.getStrategy());  // 采用策略
                map.put("initial", task.getInitial());  // 初始资金
                map.put("result", task.getResult());  // 结果
                map.put("status", task.getStatus());  // 任务状态
                map.put("createTime",task.getCreateTime());
                tasks.add(map);
            }
        }

        model.addAttribute("tasks", tasks);
        return "/site/predict";
    }

    //详情界面：
    @RequestMapping(path = "/taskResult/{taskid}", method = RequestMethod.GET)
    public String showTaskResult(@PathVariable("taskid") int taskid, Page page, Model model) {
        // 1. 从数据库中获取任务数据
        Task task = taskService.findTaskById(taskid);
        int category = task.getCategory();
       // if (task == null) {
         //   model.addAttribute("error", "Task not found");
        //    return "error";
       // }
       // if(System.currentTimeMillis()-task.getCreateTime()<30*1000){
         //   System.out.println("timediff is shorter than 30");
            //xiaoyu30s
         //   model.addAttribute("timeerror","Task not finished");
        //    return "error";

       // }
        taskService.executeScript(task);
        String result = taskService.getTaskResult(task);
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            JsonNode jsonNode = objectMapper.readTree(result);
            Map<String, Double> keyValueMap = new HashMap<>();
            jsonNode.fields().forEachRemaining(entry -> {
                String key = entry.getKey();
                Double value = entry.getValue().asDouble();
                keyValueMap.put(key, value);
            });
            if(category==1)

            {
                Double 基准收益率 = keyValueMap.get("benchmark");
                Double 账户价值 = keyValueMap.get("final_balance");
                String 日志信息 = keyValueMap.get("logInfo").toString();
                String 收益率图片 = keyValueMap.get("returnRate").toString();
                String 预测图片 = keyValueMap.get("predictResult").toString();
                Double 收益率 = keyValueMap.get("Return");

                model.addAttribute("taskId", taskid);
                model.addAttribute("category", category);
                model.addAttribute("benchmarkReturnRate", 基准收益率);
                model.addAttribute("accountValue", 账户价值);
                model.addAttribute("logInfo", 日志信息);
                model.addAttribute("returnRateImage", 收益率图片);

                model.addAttribute("predictionImage", 预测图片);

                model.addAttribute("returnRate", 收益率);
            }
            else
            {
                Double 基准收益率 = keyValueMap.get("benchmark");
                Double 账户价值 = keyValueMap.get("final_balance");
                String 日志信息 = keyValueMap.get("logInfo").toString();
                String 收益率图片 = keyValueMap.get("returnRate").toString();
                Double 收益率 = keyValueMap.get("Return");

                model.addAttribute("taskId", taskid);
                model.addAttribute("category", category);
                model.addAttribute("benchmarkReturnRate", 基准收益率);
                model.addAttribute("accountValue", 账户价值);
                model.addAttribute("logInfo", 日志信息);
                model.addAttribute("returnRateImage", 收益率图片);

                model.addAttribute("returnRate", 收益率);
            }

            return "/site/result";
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "/site/result";
    }

    @RequestMapping(path = "/addTask", method = RequestMethod.POST)
    @ResponseBody
    @LoginRequired
    public String addTask(int category, String model, String strategy, int initial, String stock, Date startDate, Date endDate ) {
        User user = hostHolder.getUser();
        if (user == null) {
            return CommunityUtil.getJSONString(403, "你还没有登录哦!");
        }
        Task task = new Task();
        task.setCategory(category);
        task.setStock(stock);
        task.setEndDate(endDate);
        task.setStartDate(startDate);
        task.setUserid(user.getId());
        task.setModel(model);
        task.setStrategy(strategy);
        task.setInitial(initial);
        task.setStatus(1);
        long totalMilisSeconds = System.currentTimeMillis();
        task.setCreateTime(totalMilisSeconds);
        taskService.addTask(task);


        //TODO:直接在这里用方法内的task信息执行python脚本,数据库填充result属性;考虑脚本执行失败的情况，并做相对应的展示界面（或者重定向至error界面）

        return CommunityUtil.getJSONString(0, "Succeeded to submit task!");

      // } else {
         //   return CommunityUtil.getJSONString(1, "Failed to execute script. Please try again.");
    }





       // return CommunityUtil.getJSONString(0, "succeed to submit task!");
    //}

}
