package com.nowcoder.community.dao;

import com.nowcoder.community.entity.Task;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface TaskMapper {
    int insertTask(Task task);
    int updateTask(Task task);
    int updateTaskStatus(int taskid, int status);

    int selectCountByUserid(int userid);
    List<Task> selectTasksByUserid(int userid, int offset, int limit);


    Task selectTaskById(int id);

    String getTaskResult(int id);

    int addTask(Task task);
}
