/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "config.h"

namespace eqMivt
{

Config::Config( eq::ServerPtr parent )
        : eq::Config( parent )
        , _redraw( true )
{
}

Config::~Config()
{
}

bool Config::init()
{
    registerObject( &_frameData );
    _frameData.setAutoObsolete( getLatency( ));

    _initData.setFrameDataID( _frameData.getID( ));
    registerObject( &_initData );

    if( !eq::Config::init( _initData.getID( )))
    {
        _deregisterData();
        return false;
    }

    return true;
}

bool Config::exit()
{
    const bool ret = eq::Config::exit();
    _deregisterData();

    return ret;
}

void Config::_deregisterData()
{
    deregisterObject( &_initData );
    deregisterObject( &_frameData );

    _initData.setFrameDataID( eq::UUID( ));
}

bool Config::loadData( const eq::uint128_t& initDataID )
{
    if( !_initData.isAttached( ))
    {
        const uint32_t request = mapObjectNB( &_initData, initDataID,
                                              co::VERSION_OLDEST,
                                              getApplicationNode( ));
        if( !mapObjectSync( request ))
            return false;
        unmapObject( &_initData ); // data was retrieved, unmap immediately
    }
    else // appNode, _initData is registered already
    {
        LBASSERT( _initData.getID() == initDataID );
    }
    return true;
}

uint32_t Config::startFrame()
{
    const eq::uint128_t& version = _frameData.commit();

    _redraw = false;
    return eq::Config::startFrame( version );
}

bool Config::handleEvent( eq::EventICommand command )
{
    switch( command.getEventType( ))
    {
        case eq::Event::KEY_PRESS:
        {
	    const eq::Event& event = command.get< eq::Event >();
            if( _handleKeyEvent( event.keyPress ))
            {
                _redraw = true;
                return true;
            }
            break;
        }

        case eq::Event::CHANNEL_POINTER_BUTTON_PRESS:
        {
	    break;
        }

        case eq::Event::CHANNEL_POINTER_BUTTON_RELEASE:
        {
            break;
        }
        case eq::Event::CHANNEL_POINTER_MOTION:
        {
	    const eq::Event& event = command.get< eq::Event >();
            switch( event.pointerMotion.buttons )
            {
              case eq::PTR_BUTTON1:

                      _frameData.spinCamera(
                          -0.005f * event.pointerMotion.dy,
                          -0.005f * event.pointerMotion.dx );
                  _redraw = true;
                  return true;

              case eq::PTR_BUTTON2:
                  _frameData.moveCamera( 0.f, 0.f, .005f );
                  _redraw = true;
                  return true;

              case eq::PTR_BUTTON3:
                  _frameData.moveCamera(  .0005f * event.pointerMotion.dx,
                                         -.0005f * event.pointerMotion.dy,
                                          0.f );
                  _redraw = true;
                  return true;
            }
            break;
        }

        case eq::Event::CHANNEL_POINTER_WHEEL:
        {
	    const eq::Event& event = command.get< eq::Event >();
            _frameData.moveCamera( -0.05f * event.pointerWheel.yAxis,
                                   0.f,
                                   0.05f * event.pointerWheel.xAxis );
            _redraw = true;
            return true;
        }

        case eq::Event::MAGELLAN_AXIS:
        {
	    break;
        }

        case eq::Event::MAGELLAN_BUTTON:
        {
	    break;
        }

        case eq::Event::WINDOW_EXPOSE:
        case eq::Event::WINDOW_RESIZE:
        case eq::Event::WINDOW_CLOSE:
        case eq::Event::VIEW_RESIZE:
            _redraw = true;
            break;
    default:
        break;
    }

    _redraw |= eq::Config::handleEvent( command );
    return _redraw;
}

bool Config::_handleKeyEvent( const eq::KeyEvent& event )
{
    switch( event.key )
    {
        default:
            return false;
    }
}

bool Config::needRedraw()
{
    return _redraw;
}

co::uint128_t Config::sync( const co::uint128_t& version )
{
    return eq::Config::sync( version );
}
}
