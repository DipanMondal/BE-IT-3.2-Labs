<%@ page import="java.sql.*, javax.sql.*" %>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1" pageEncoding="ISO-8859-1"%>

<%
    String loginName = request.getParameter("login_name");
    String password = request.getParameter("password");
    String fullName = request.getParameter("full_name");
    String email = request.getParameter("email");

    Connection conn = null;
    PreparedStatement stmt = null;

    String dbURL = "jdbc:mysql://172.16.4.234:3306/test";
    String dbUsername = "guest";  // Replace with your MySQL username
    String dbPassword = "guest";  // Replace with your MySQL password

    try {
        // Set up the database connection
        Class.forName("com.mysql.jdbc.Driver");
        conn = DriverManager.getConnection(dbURL, dbUsername, dbPassword);

        // Insert user data into the users table
        String query = "INSERT INTO users1 (login_name, password, full_name, email) VALUES (?, ?, ?, ?)";
        stmt = conn.prepareStatement(query);
        stmt.setString(1, loginName);
        stmt.setString(2, password);  // Consider hashing the password before storing it
        stmt.setString(3, fullName);
        stmt.setString(4, email);

        int result = stmt.executeUpdate();

        if (result > 0) {
            out.print("Account created successfully.");
        } else {
            out.print("Error creating account.");
        }
    } catch (Exception e) {
        e.printStackTrace();
    } finally {
        try {
            if (stmt != null) stmt.close();
            if (conn != null) conn.close();
        } catch (SQLException se) {
            se.printStackTrace();
        }
    }
%>
