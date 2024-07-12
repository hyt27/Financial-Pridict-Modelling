package com.nowcoder.community.config;


import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.thymeleaf.spring5.SpringTemplateEngine;
import org.thymeleaf.spring5.view.ThymeleafViewResolver;
import java.util.Locale;

@Configuration
public class ThymeleafConfig {

    @Bean
    public ThymeleafViewResolver thymeleafViewResolver(SpringTemplateEngine templateEngine) {
        ThymeleafViewResolver resolver = new ThymeleafViewResolver() {
            @Override
            public org.springframework.web.servlet.View resolveViewName(String viewName, Locale locale) throws Exception {
                if (viewName != null && viewName.startsWith("/")) {
                    viewName = viewName.substring(1);
                }
                return super.resolveViewName(viewName, locale);
            }
        };
        resolver.setTemplateEngine(templateEngine);
        resolver.setCharacterEncoding("UTF-8");
        resolver.setOrder(0);  // 设置为高优先级
        return resolver;
    }
}

