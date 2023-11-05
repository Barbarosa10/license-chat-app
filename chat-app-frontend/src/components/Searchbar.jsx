import React from 'react'

const Searchbar = () => {
    return(
        <div className="search">
            <div className='searchForm'>
                <input type="text" placeholder='Search for an user'/>
            </div>
            <div className='userConversation'>
                <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" />
                <div className='userConversationInfo'>
                    <span>Duciuc Danut</span>
                </div>
            </div>
        </div>
    )
}

export default Searchbar