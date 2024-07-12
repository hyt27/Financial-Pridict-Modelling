package com.nowcoder.community.service;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nowcoder.community.dao.TaskMapper;
import com.nowcoder.community.entity.Task;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class TaskService {
    private static final Logger logger = LoggerFactory.getLogger(TaskService.class);


    @Autowired
    private TaskMapper taskMapper;
    private ObjectMapper objectMapper = new ObjectMapper();

    public int findTaskCount(int userid) {
        return taskMapper.selectCountByUserid(userid);
    }
    public List<Task> findTasksByUserId( int userId, int offset, int limit) {
        return taskMapper.selectTasksByUserid(userId, offset, limit);
    }
    public Task findTaskById( int id) {
        return taskMapper.selectTaskById(id);
    }


    public Map<String, Object> addTask(Task task) {
        Map<String, Object> map = new HashMap<>();

        // 空值处理
        if (task == null) {
            throw new IllegalArgumentException("task不能为空!");
        }
        if (task.getCategory()==0) {
            map.put("categoryMsg", "category不能为空!");
            return map;
        }
        if (StringUtils.isBlank(task.getStock())) {
            map.put("StockMsg", "stock不能为空!");
            return map;
        }
        if (StringUtils.isBlank(task.getStartDate().toString())) {
            map.put("startDateMsg", "start_Date不能为空!");
            return map;
        }
        if (StringUtils.isBlank(task.getEndDate().toString())) {
            map.put("endDateMsg", "end_Date不能为空!");
            return map;
        }

        if (StringUtils.isBlank(task.getModel())) {
            map.put("modelMsg", "model不能为空!");
            return map;
        }
        if (StringUtils.isBlank(task.getStrategy())) {
            map.put("strategyMsg", "strategy不能为空!");
            return map;
        }
        if (task.getInitial()==0) {
            map.put("initialMsg", "initial不能为空!");
            return map;
        }

        taskMapper.insertTask(task);
        return map;
    }
    public int markTaskAsDeleted(Task task){
        if(task==null){
            throw new IllegalArgumentException("task不能为空!");
        }
        int rows= taskMapper.updateTaskStatus(task.getId(),-1);
        return rows;
    }

    public boolean  updateTask(Task task){
        if (task == null ) {
            logger.error("Attempted to update a task with null task or task ID.");

            throw new IllegalArgumentException("Task and Task ID must not be null for update.");
        }
        try {
            int rows = taskMapper.updateTask(task);
            return rows > 0;
        } catch (Exception e) {
            logger.error("Error updating task: " + e.getMessage(), e);
            throw new RuntimeException("Failed to update task", e);
        }

    }

    public String executeScript(Task task){
        //TODO： 运行脚本得到结果；这个服务在TaskController的addTask方法内被调用
        int id = task.getId();
        int category = task.getCategory();
        String stock = task.getStock();
        Date start = task.getStartDate();
        Date end = task.getEndDate();
        String model = task.getModel();
        String Strategy = task.getStrategy();
        int initial = task.getInitial();
        SimpleDateFormat outputDateFormat = new SimpleDateFormat("yyyy-M-d");
        String formattedStartDate = outputDateFormat.format(start);
        String formattedEndDate = outputDateFormat.format(end);
        try {
            // 构建命令
            String projectRoot = new File("").getAbsolutePath();
            // 构建相对路径
            String scriptPath = projectRoot + File.separator + "src" + File.separator + "main" + File.separator + "resources" + File.separator + "6.17_predict"+  File.separator +"interface.py";
            String pythonPath = "python"; // 或者使用Python解释器的完整路径
            String[] command = new String[]{pythonPath, scriptPath, String.valueOf(id),String.valueOf(category),stock,formattedStartDate,formattedEndDate,model,Strategy,String.valueOf(initial)};
            System.out.println("Executing command: " + String.join(" ", command));
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorOutput = new StringBuilder();
            while ((line = errorReader.readLine()) != null) {
                errorOutput.append(line);
            }
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("Python脚本执行失败，退出代码：" + exitCode + "\n错误信息：" + errorOutput.toString());
            }
            String inputString = output.toString();
            int startIndex = inputString.indexOf("{");
            int endIndex = inputString.indexOf("}");
            String result = inputString.substring(startIndex,endIndex+1);
            System.out.println("Python Output: " + result);
            JSONObject jsonObject = JSONObject.parseObject(result);
            String taskResult = jsonObject.toJSONString();
            task.setResult(taskResult);
            taskMapper.updateTask(task);

            return taskResult;
        } catch (Exception e) {
            throw new RuntimeException("调用Python脚本失败", e);
        }
        //return  "1";

    }


    public String getTaskResult(Task task) {
        //TODO:移动这里的《运行python脚本》至executeScript方法
        // 这里只负责从数据库中获取result属性
        // 我在task controller中添加了判断时间逻辑，若不足三十秒跳转到（错误页）（还可以精进一下，之后再考虑）
        // 若通过时间检查，则应去找对应结果并渲染在结果详情页；
        if (task == null) {
            throw new IllegalArgumentException("task不能为空!");
        }
        int id = task.getId();
        String taskResult = taskMapper.getTaskResult(id);
        return taskResult;

    }
}
