<%@ page language="java" contentType="text/html; charset=ISO-8859-1" pageEncoding="ISO-8859-1"%>
<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>

<!DOCTYPE html>
<html>
<head>
    <title>Open Account</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        // Function to check if login name is available
        function checkLoginAvailability() {
            var loginName = $('#login_name').val();
            
            if (loginName.length > 0) {
                $.ajax({
                    url: "checkLoginAvailability.jsp",
                    type: "GET",
                    data: { login_name: loginName },
                    success: function(response) {
                        $('#availability_status').html(response);
                    }
                });
            } else {
                $('#availability_status').html("");
            }
        }
    </script>
</head>
<body>

    <h2>Open Account</h2>

    <form action="insertAccount.jsp" method="post">
        <label for="login_name">Login Name:</label>
        <input type="text" id="login_name" name="login_name" onkeyup="checkLoginAvailability()" required>
        <span id="availability_status"></span><br><br>

        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required><br><br>

        <label for="full_name">Full Name:</label>
        <input type="text" id="full_name" name="full_name"><br><br>

        <label for="email">Email:</label>
        <input type="email" id="email" name="email"><br><br>

        <input type="submit" value="Create Account">
    </form>

</body>
</html>
