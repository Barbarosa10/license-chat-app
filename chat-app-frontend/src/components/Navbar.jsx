import React from 'react'
import profileImage from "../images/addAvatar.png";

const Navbar = () => {
    return(
        <div className='navbar'>
            <div className='logo'>
                <span>Chat App</span>
            </div>

            <div className='user'>
                <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" />
                <span>Duciuc Danut</span>
                <button>Logout</button>
            </div>
        </div>
    )
}

export default Navbar