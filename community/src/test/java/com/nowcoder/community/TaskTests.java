package com.nowcoder.community;

import com.nowcoder.community.entity.Task;
import com.nowcoder.community.service.TaskService;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.Date;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(SpringRunner.class)
@SpringBootTest
@ContextConfiguration(classes = CommunityApplication.class)
public class TaskTests {
    @Autowired
    private TaskService taskService;

    @Test
    public void testAddTask() {
        // 创建一个新的任务对象
        Task task = new Task();
        task.setCategory(2);
        task.setStock("AAPL");
        task.setStartDate(new Date());
        task.setEndDate(new Date());
        task.setUserid(154);
        task.setModel("Default");
        task.setStrategy("LSTM");
        task.setInitial(10000);
        task.setStatus(0);
        task.setCreateTime(System.currentTimeMillis());

        // 尝试添加任务
        taskService.addTask(task);
        System.out.println("Task added successfully.");
    }

    @Test
    public void testUpdateTaskStatus() {
        // 假设有一个已知的任务ID
        int taskId = 1; // 假设存在的任务ID
        Task task = new Task();
        task.setId(taskId);
        task.setStatus(-1); // 标记为已删除

        // 更新任务状态
        taskService.markTaskAsDeleted(task);
        System.out.println("Task status updated to deleted.");
    }

    @Test
    public void testSelectTaskById(){
        Task task = taskService.findTaskById(6);
        System.out.println(task.toString());
    }

    @Test
    public void testGetTaskResult() {
        Task task = taskService.findTaskById(4);
        //System.out.println(task.toString());
        String taskResult = taskService.getTaskResult(task);
        task.setResult(taskResult);

        taskService.updateTask(task);
        System.out.println("Task result has been updated and added into DB .");

    }


}
