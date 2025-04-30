import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.sql.*;
import java.util.*;
import com.google.gson.Gson;
import java.sql.Connection;
import java.sql.DriverManager;

class JDBCUtil {
    public static Connection getConnection() throws Exception {
        Class.forName("com.mysql.jdbc.Driver");
        return DriverManager.getConnection(
            "jdbc:mysql://172.16.4.234:3306/test", "guest", "guest"
        );
    }
}

public class GetDepartments extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
    throws ServletException, IOException {
        response.setContentType("application/json");
        try (Connection con = JDBCUtil.getConnection()) {
            Statement stmt = con.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT DISTINCT dept_name FROM students112");
            List<String> departments = new ArrayList<>();
            while (rs.next()) {
                departments.add(rs.getString("dept_name"));
            }
            String json = new Gson().toJson(departments);
            response.getWriter().write(json);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
