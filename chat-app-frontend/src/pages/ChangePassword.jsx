import React from "react";

const ChangePassword = () => {
    return(
        <div className="formContainer">
            <div className="formWrapper">
                <span className="title">Change Password</span>
                <form action="login">
                    <input type="password" placeholder="New Password"/>
                    <input type="password" placeholder="Retype Password"/>
                    <button>Change Password</button>
                </form>
            </div>
        </div>
    )
}

export default ChangePassword