package com.nowcoder.community.repository;

import com.nowcoder.community.entity.UserStock;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserStockRepository extends JpaRepository<UserStock, Integer> {
}
