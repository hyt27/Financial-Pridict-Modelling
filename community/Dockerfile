# 基于Java环境作为基础镜像
FROM openjdk:17-jdk

## 使用官方MySQL镜像作为基础
#FROM mysql:latest

# 指定容器内的工作目录
WORKDIR /app

# 将构建产物复制到容器内
COPY target/community-0.0.1-SNAPSHOT.jar /app/community-0.0.1-SNAPSHOT.jar

# 暴露容器的8080端口
EXPOSE 8080

# 容器启动时执行的命令
ENTRYPOINT ["java", "-jar", "/app/community-0.0.1-SNAPSHOT.jar"]



## 设置环境变量，例如MySQL的root密码
#ENV MYSQL_ROOT_PASSWORD=xzh830600
## 将您的init.sql脚本复制到容器中的/docker-entrypoint-initdb.d/目录
## 这个目录中的脚本会在MySQL容器启动时自动执行
#COPY ./init.sql /docker-entrypoint-initdb.d/
#
## 暴露MySQL默认端口
#EXPOSE 3306
#
## 设置持久化存储，以便在容器重启后保留数据
#VOLUME /var/lib/mysql
