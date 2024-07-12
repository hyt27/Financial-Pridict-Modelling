package com.nowcoder.community.dao;

import com.nowcoder.community.entity.UserStock;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface UserStockMapper {

    @Insert("INSERT INTO user_stocks (user_id, stock_id) VALUES (#{userId}, #{stockId})")
    void insertUserStock(UserStock userStock);

    @Delete("DELETE FROM user_stocks WHERE user_id = #{userId} AND stock_id = #{stockId}")
    void deleteUserStock(int userId, int stockId);

    @Select("SELECT COUNT(*) FROM user_stocks WHERE user_id = #{userId} AND stock_id = #{stockId}")
    int countUserStock(int userId, int stockId);

    @Select("SELECT * FROM user_stocks WHERE user_id = #{userId}")
    List<UserStock> selectUserStocksByUserId(int userId);
}

