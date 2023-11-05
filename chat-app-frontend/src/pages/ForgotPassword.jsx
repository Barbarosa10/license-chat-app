import React from "react";

const ForgotPassword = () => {
    return(
        <div className="formContainer">
            <div className="formWrapper">
                <span className="title">Change Password</span>
                <form>
                    <input type="email" placeholder="Email"/>
                    <button>Send mail</button>
                    <input type="text"  placeholder="Type the code from mail"/>
                    <button formAction="changePassword">Check code</button>
                </form>
            </div>
        </div>
    )
}

export default ForgotPassword