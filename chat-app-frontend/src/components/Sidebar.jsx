import React from 'react'
import Navbar from "./Navbar";
import Searchbar from './Searchbar';
import Conversations from './Conversations';

const SideBar = ({conversations}) => {


    return(
        <div className='sidebar'>
            <Navbar/>
            <Searchbar/>
            <Conversations conversations={conversations}/>
        </div>
    )
}

export default SideBar