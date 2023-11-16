import React from 'react';
import { useNavigate, Link } from "react-router-dom";
import profileImage from "../images/addAvatar.png";

const localhost_key = "chat-app-current-user"

const Navbar = () => {
    const navigate = useNavigate();

    const logout = () => {
        localStorage.clear(localhost_key)
        navigate("/login");
    }

    return(
        <div className='navbar'>
            <div className='logo'>
                <span>Chat App</span>
            </div>

            <div className='user'>
                <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" />
                <span>Duciuc Danut</span>
                <button onClick={logout}>Logout</button>
            </div>
        </div>
    )
}

export default Navbar