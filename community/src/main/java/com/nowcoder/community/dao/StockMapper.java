package com.nowcoder.community.dao;

import com.nowcoder.community.entity.Stock;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface StockMapper {

    @Select("SELECT id, name, code FROM stocks")
    List<Stock> selectAllStocks();

    @Select("SELECT id, name, code FROM stocks WHERE code = #{code}")
    Stock selectByCode(String code);

    @Select("SELECT id, name, code FROM stocks WHERE id = #{id}")
    Stock selectById(int id);
}

